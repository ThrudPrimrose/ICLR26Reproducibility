subroutine tsvc_2_s13110_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(in)  :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(2, 2)
  character(kind=c_char), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: l, ii, jj, jstar, istar
  real(c_double) :: maxv, m, chksum
  real(c_double), allocatable :: colmax(:)

  l = len_2d
  allocate(colmax(l))

  ! numpy aa(i,j) = Fortran aa(j+1, i+1): a numpy ROW is a Fortran COLUMN.
  ! The reference's row-major first max is therefore: the first Fortran column
  ! jj that contains the max, then the first row ii inside that column.
  ! One full read of aa: per-column maxima (contiguous, SIMD, threaded).
  !$omp parallel do schedule(static)
  do jj = 1, l
    m = -huge(0d0)
    do ii = 1, l
      m = max(m, aa(ii,jj))
    end do
    colmax(jj) = m
  end do

  maxv = maxval(colmax)
  jstar = 1
  do jj = 1, l
    if (colmax(jj) == maxv) then
      jstar = jj
      exit
    end if
  end do

  istar = 1
  do ii = 1, l
    if (aa(ii,jstar) == maxv) then
      istar = ii
      exit
    end if
  end do

  ! xindex = numpy row i = jstar-1 ; yindex = numpy column j = istar-1
  chksum = maxv + dble(jstar-1) + dble(istar-1)
  bb(1,1) = chksum
  deallocate(colmax)
end subroutine tsvc_2_s13110_fp64
