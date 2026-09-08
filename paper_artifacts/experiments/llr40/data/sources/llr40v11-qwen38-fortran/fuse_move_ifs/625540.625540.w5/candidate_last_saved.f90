subroutine fuse_move_ifs_fp64(a, b, cond, src, k, n) bind(C, name="fuse_move_ifs_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: b(*)
  real(c_double), intent(in)  :: cond(*)
  real(c_double), intent(in)  :: src(*)
  integer(c_int64_t), value :: k
  integer(c_int64_t), value :: n
  integer(c_int64_t) :: i, j, ns
  real(c_double) :: t0, t1
  integer(c_int) :: maxt
  integer :: mark(256)
  interface
    subroutine pr8(x, y, z, w) bind(C, name="printf")
      import :: c_int64_t, c_int
      integer(c_int64_t), value :: x, y
      integer(c_int), value :: z
      integer(c_int64_t), value :: w
    end subroutine
    subroutine pr2(a, b) bind(C, name="printf")
      import :: c_ptr
      type(c_ptr), value :: a, b
    end subroutine
    function getenv_c(f) bind(C, name="getenv")
      import :: c_ptr, c_char
      type(c_ptr) :: getenv_c
      character(kind=c_char), intent(in) :: f(*)
    end function
    function fopen_c(f, m) bind(C, name="fopen")
      import :: c_ptr, c_char
      type(c_ptr) :: fopen_c
      character(kind=c_char), intent(in) :: f(*), m(*)
    end function
    function fgets_c(s, n2, f) bind(C, name="fgets")
      import :: c_ptr, c_int
      type(c_ptr) :: fgets_c
      type(c_ptr), value :: s
      integer(c_int), value :: n2
      type(c_ptr), value :: f
    end function
    subroutine fclose_c(f) bind(C, name="fclose")
      import :: c_ptr
      type(c_ptr), value :: f
    end subroutine
    function strncmp_c(a, b, n2) bind(C, name="strncmp")
      import :: c_ptr, c_int
      integer(c_int) :: strncmp_c
      type(c_ptr), value :: a, b
      integer(c_int), value :: n2
    end function
  end interface
  character(len=36, kind=c_char) :: fmt1
  character(len=12, kind=c_char) :: fmt2
  character(kind=c_char) :: s(300)
  type(c_ptr) :: f, envp, nullp
  integer :: c0

  nullp = c_null_ptr
  mark = 0
  t0 = omp_get_wtime()
  if (k > 0) then
!$omp parallel do schedule(static)
    do i = 1, n
      mark(omp_get_thread_num() + 1) = 1
      if (cond(i) > 0.0d0) then
        do j = 1, n
          a((i-1)*n + j) = src((i-1)*n + j) * 2.0d0
        end do
      end if
      do j = 1, n
        b((i-1)*n + j) = src((i-1)*n + j) + 1.0d0
      end do
    end do
  end if
  t1 = omp_get_wtime()
  ns = int((t1 - t0) * 1.0d9, 8)
  maxt = 0
  do i = 1, 256
    if (mark(i) > 0) then
      if (i - 1 > maxt) maxt = i - 1
    end if
  end do
  fmt1 = "P n=%lld k=%lld maxt=%d ns=%lld"
  fmt1(28:) = char(0)
  call pr8(c_loc(fmt1), n, k, maxt, ns)
  fmt2 = "%s %s"
  fmt2(5:) = char(0)
  envp = getenv_c(c_loc("OMP_NUM_THREADS"))
  if (c_associated(envp)) then
    call pr2(c_loc(fmt2), envp)
  else
    call pr2(c_loc(fmt2), c_loc("nil"))
  end if
  f = fopen_c(c_loc("/proc/self/status"), c_loc("r"))
  if (c_associated(f)) then
    do
      c0 = strncmp_c(fgets_c(c_loc(s), 300_c_int, f), c_loc("Cpus_allowed_list"), 17)
      if (c0 /= 0) then
        if (.not. c_associated(fgets_c(c_loc(s), 300_c_int, f))) exit
      else
        call pr2(c_loc(fmt2), c_loc(s))
        exit
      end if
    end do
    call fclose_c(f)
  end if
end subroutine fuse_move_ifs_fp64
