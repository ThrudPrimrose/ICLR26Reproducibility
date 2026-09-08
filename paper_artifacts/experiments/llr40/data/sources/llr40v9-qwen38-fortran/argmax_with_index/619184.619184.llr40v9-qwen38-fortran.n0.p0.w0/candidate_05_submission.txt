subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(c, name='argmax_with_index_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(in), dimension(*) :: a
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), value, intent(in) :: len_1d
  integer, external :: omp_get_max_threads
  real(c_double) :: x, gmax=-huge(1.0d0), lmax(512), blk(8), bmax
  integer(c_int64_t) :: i, idx, gidx, lidx(512), n, nt, t, start, fin, chunk, base, n8, j, pos
  logical :: has(512)
  if (len_1d <= 0) then
    out_value = 0.0d0
    out_index = 0
    return
  end if
  n = len_1d
  nt = omp_get_max_threads()
  if (nt > 512) nt = 512
  if (nt < 1) nt = 1
  chunk = (n + nt - 1) / nt
  !$omp parallel do schedule(static) private(t,start,fin,x,idx,blk,bmax,base,n8,j,pos)
  do t = 0, nt-1
    start = t * chunk
    fin = min(n, start + chunk)
    if (start < fin) then
      has(t+1) = .true.
      x = -huge(1.0d0)
      idx = -1
      n8 = (fin - start) / 8
      do i = 0, n8-1
        base = start + i*8
        blk(1)=a(base+1); blk(2)=a(base+2); blk(3)=a(base+3); blk(4)=a(base+4)
        blk(5)=a(base+5); blk(6)=a(base+6); blk(7)=a(base+7); blk(8)=a(base+8)
        bmax = blk(1)
        do j = 2, 8
          if (blk(j) > bmax) then; bmax = blk(j); end if
        end do
        if (bmax > x) then
          x = bmax
          pos = 0
          do j = 1, 8
            if (blk(j) == bmax) then; pos = j - 1; exit; end if
          end do
          idx = base + pos
        end if
      end do
      do i = start + n8*8, fin - 1
        if (a(i+1) > x) then; x = a(i+1); idx = i; end if
      end do
      lmax(t+1) = x
      lidx(t+1) = idx + 1
    else
      has(t+1) = .false.
    end if
  end do
  !$omp end parallel do
  gidx = 0
  do t = 0, nt-1
    if (has(t+1)) then
      if (gidx == 0) then
        gmax = lmax(t+1); gidx = lidx(t+1)
      else if (lmax(t+1) > gmax) then
        gmax = lmax(t+1); gidx = lidx(t+1)
      end if
    end if
  end do
  out_value = gmax
  out_index = gidx
end subroutine
